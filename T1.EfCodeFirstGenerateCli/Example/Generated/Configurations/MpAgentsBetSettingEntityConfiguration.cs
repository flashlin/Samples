using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpAgentsBetSettingEntityConfiguration : IEntityTypeConfiguration<MpAgentsBetSettingEntity>
    {
        public void Configure(EntityTypeBuilder<MpAgentsBetSettingEntity> builder)
        {
            builder.ToTable("MpAgentsBetSetting");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ParentId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Credit)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.MaxCredit)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.MinimumBet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.MaximumBet)
                .HasColumnType("decimal(19,2)")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

        }
    }
}
