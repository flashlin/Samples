using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SB_RobotBetQuarantineLogEntityConfiguration : IEntityTypeConfiguration<SB_RobotBetQuarantineLogEntity>
    {
        public void Configure(EntityTypeBuilder<SB_RobotBetQuarantineLogEntity> builder)
        {
            builder.ToTable("SB_RobotBetQuarantineLog");


            builder.Property(x => x.custId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
