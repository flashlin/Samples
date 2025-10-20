using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentFeatureEntityConfiguration : IEntityTypeConfiguration<AgentFeatureEntity>
    {
        public void Configure(EntityTypeBuilder<AgentFeatureEntity> builder)
        {
            builder.ToTable("AgentFeature");

            builder.HasKey(x => new { x.CustID, x.RoleID });

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.RoleID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.OTPEmail)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.RecoveryEmail)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Secret)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.OTPNotRemindMe)
                .HasColumnType("bit")
            ;

        }
    }
}
